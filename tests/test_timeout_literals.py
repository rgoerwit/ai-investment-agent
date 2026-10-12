"""No literal deadline at an execution boundary outside the config owners.

Why this exists
---------------
Before October 2026 about 75 timeout values were scattered across ``src/`` as
literals: a vendor ``ClientTimeout(total=10)``, a ``run_with_hard_timeout(...,
timeout=30)``, a client built with ``"timeout": 240``. No override and no caller
deadline could reach them, and the same number drifted between copies. They now
live in ``Settings`` (or a named policy constant whose parity with its setting is
tested) and are clamped to the caller's enclosing deadline at the call.

This guard keeps it that way. It flags a numeric literal reaching a deadline
parameter -- ``timeout=``, ``*_timeout=``, ``*_timeout_seconds=``,
``ClientTimeout``/``Timeout`` fields, ``BlockingCallPolicy``'s timeout,
``wait_for``'s timeout, a ``"timeout"`` key in a kwargs dict, or a deadline
parameter's own default. A literal is followed through a module constant, a
local variable, ``x or 240``, ``a if c else 240``, ``min(x, 30)``, ``max(x, 30)``
and ``x + 30``. ``max(x, 0)`` is a positivity floor and ``x * 0.5`` a scale, so
neither counts; a product counts only when both sides are literal. Names follow
lexical scope: a loop, ``with``, ``except`` or unpacking target shadows a module
constant, and a nested def's locals stay its own.

Three further routes are followed, from an index of the whole tree. A name
imported from another module (``from m import T [as U]``, ``m.T``, re-export
hops) resolves to that module's constant, through chains such as ``B = A + 5``,
config owners included: the owner is exempt as a scan target, not as a source
of literals. A three-argument ``getattr(obj, "field", 30)`` counts as its
fallback. And any function defined in ``src/`` with a positional or
keyword-only parameter shaped like a deadline (``timeout``, ``seconds``,
``deadline`` and their ``_seconds`` forms) is a deadline callee, found by that
signature and not by a list of names. A call is matched to the def its name is
bound to, import aliases included (``import enclosing_deadline as ed``, then
``ed(240)``), and a same-named unrelated function is not matched. Residual holes: a method on
a receiver the AST cannot type matches every src method of that name, and a src
function that *returns* a literal is not followed.

It is **not** a ban on every ``TIMEOUT`` or ``RETRY`` constant: a constant that
never reaches a deadline parameter is not its business, and the config owners
below are exempt. Polling intervals, cooldowns and protocol constants are out of
scope; the few that share a deadline's spelling are allowlisted with that reason.

Allowlist entries are keyed ``path::scope::site`` -- the enclosing function's
qualified name (or, at module level, the assignment target) and the parameter --
not a line number, so an unrelated edit cannot make an entry stale. An entry no
longer found fails the stale test, so the list cannot outlive its code.
"""

from __future__ import annotations

import ast
import functools
import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"

# The homes that own default numbers (see .claude/rules/config-precedence.md).
CONFIG_OWNERS: dict[str, str] = {
    "src/config.py": "Settings: the env-driven defaults themselves",
    "src/runtime_config.py": "RuntimeConfig: run-scoped fields a CLI flag overrides",
    "src/ibkr_config.py": "broker settings with their own env overrides",
    "src/ibkr/portfolio_defaults.py": "canonical portfolio knob defaults",
    "src/llm_runtime/seats.py": "SeatExecutionPolicy: per-seat client timeouts are seat data",
    "src/web/ibkr_dashboard/settings.py": "the dashboard's own settings model",
}

# Each entry: "path::scope::site" -> why a literal is right here.
ALLOWLIST: dict[str, str] = {
    # Named BlockingCallPolicy defaults. Each is only a construction default:
    # every call site re-sizes it with .with_timeout(<setting>) clamped to the
    # caller's deadline, and a parity test pins the number to its setting.
    "src/blocking_io.py::YFINANCE_INFO_POLICY::BlockingCallPolicy[1]": (
        "parity with company_name_lookup_timeout_seconds "
        "(tests/data/test_market_data_timeouts.py); re-sized at the call"
    ),
    "src/blocking_io.py::YAHOOQUERY_QUOTE_TYPE_POLICY::BlockingCallPolicy[1]": (
        "parity with company_name_lookup_timeout_seconds "
        "(tests/data/test_market_data_timeouts.py); re-sized at the call"
    ),
    "src/blocking_io.py::GAP_FILL_YFINANCE_INFO_POLICY::BlockingCallPolicy[1]": (
        "parity with company_name_lookup_timeout_seconds "
        "(tests/data/test_market_data_timeouts.py); re-sized at the call"
    ),
    "src/blocking_io.py::FX_RATE_POLICY::BlockingCallPolicy[1]": (
        "parity with fx_rate_fetch_timeout_seconds "
        "(tests/data/test_market_data_timeouts.py); re-sized at the call"
    ),
    "src/blocking_io.py::OUTPUT_COMPANY_NAME_POLICY::BlockingCallPolicy[1]": (
        "parity with output_company_name_timeout_seconds "
        "(tests/test_operational_timeouts.py); re-sized at the call"
    ),
    "src/blocking_io.py::OUTPUT_COMPANY_NAME_INFO_POLICY::BlockingCallPolicy[1]": (
        "parity with company_name_lookup_timeout_seconds "
        "(tests/test_operational_timeouts.py); re-sized at the call"
    ),
    "src/tavily_utils.py::_TAVILY_SYNC_POLICY::BlockingCallPolicy[1]": (
        "parity with tavily_search_timeout_seconds "
        "(tests/tools/test_research_timeouts.py); re-sized at the call"
    ),
    "src/tools/official_documents.py::_PDF_POLICY::BlockingCallPolicy[1]": (
        "parity with official_document_extract_timeout_seconds "
        "(tests/tools/test_research_timeouts.py); re-sized at the call"
    ),
    "src/tools/official_documents.py::_DNS_POLICY::BlockingCallPolicy[1]": (
        "parity with official_document_dns_timeout_seconds "
        "(tests/tools/test_research_timeouts.py); re-sized at the call"
    ),
    # Structural offsets and placeholders, not tunable deadlines.
    "src/agents/consultant_nodes.py::create_consultant_node::from_settings(total_timeout)": (
        "a shape-only placeholder (0.0); the real total is stamped per "
        "invocation from consultant_*_total_timeout_seconds"
    ),
    "src/agents/consultant_nodes.py::create_consultant_node::from_settings(deadline)": (
        "the same shape-only placeholder (0.0) as total_timeout above; the real "
        "deadline is stamped per invocation from the shared deadline clock"
    ),
    "src/agents/circuit_breaker.py::LLMCircuitBreaker.before_call::CircuitOpenError(opens_remaining_seconds)": (
        "0.0 reports a half-open breaker's remaining cool-off in an error; "
        "nothing waits on it, so it bounds no work"
    ),
    "src/agents/network_breaker.py::NetworkBreaker.before_call::NetworkBreakerOpenError(opens_remaining_seconds)": (
        "0.0 reports a half-open breaker's remaining cool-off in an error; "
        "nothing waits on it, so it bounds no work"
    ),
    # Defaults of deadline-shaped parameters that size no work.
    "src/agents/circuit_breaker.py::LLMCircuitBreaker.__init__::def(window_seconds)": (
        "the failure-counting window of a circuit breaker, not a deadline on a call"
    ),
    "src/agents/circuit_breaker.py::LLMCircuitBreaker.__init__::def(cool_off_seconds)": (
        "the breaker's cool-off after opening (cooldowns are out of scope)"
    ),
    "src/agents/network_breaker.py::NetworkBreaker.__init__::def(window_seconds)": (
        "the failure-counting window of a circuit breaker, not a deadline on a call"
    ),
    "src/agents/network_breaker.py::NetworkBreaker.__init__::def(cool_off_seconds)": (
        "the breaker's cool-off after opening (cooldowns are out of scope)"
    ),
    "src/async_utils.py::get_hard_timeout_orphan_snapshot::def(min_age_seconds)": (
        "the same diagnostic age filter as the run_analysis entry below"
    ),
    "src/web/ibkr_dashboard/worker.py::main::def(poll_interval_seconds)": (
        "the job worker's polling cadence (out of scope), not a deadline"
    ),
    "src/main.py::run_analysis::get_hard_timeout_orphan_snapshot(min_age_seconds)": (
        "a minimum age for listing orphaned tasks in a shutdown diagnostic, "
        "a filter on a snapshot and not a deadline on any work"
    ),
    "src/mcp/client.py::MCPRuntime._on_call_failure::_register_cooldown(duration_seconds)": (
        "a named auth/rate-limit cooldown constant: a circuit cool-off window "
        "after a failure (cooldowns are out of scope), not a deadline on a call"
    ),
    "src/agents/consultant_nodes.py::create_consultant_node.consultant_node::run_with_hard_timeout(timeout)": (
        "the settings-derived consultant total plus a fixed 30 s grace "
        "(CONSULTANT_HARD_TIMEOUT_GRACE_SECONDS): the gap that lets the "
        "cooperative deadline win over the hard node bound, a relation not a knob"
    ),
    "src/llm_runtime/construction.py::_build_legacy_client::create_gemini_model(timeout)": (
        "fallback mirrors the Health Check seat's client_timeout_seconds "
        "(10.0, src/llm_runtime/seats.py), which always sets it; legacy path only"
    ),
    # Not on the analysis path, or not a deadline at all.
    "src/eval/git_meta.py::_git_output::run_program_sync(timeout)": (
        "local git subprocess for provenance metadata (the run fingerprint "
        "reads it once per process): no network; a setting would be dead "
        "tuning surface"
    ),
    "src/ibkr/analysis_index.py::load_latest_analyses._heartbeat_worker::wait(timeout)": (
        "Event.wait used as a 30 s heartbeat interval (a polling cadence, "
        "out of scope), not a deadline on any work"
    ),
    "src/tools/shared.py::<module>::_DDGS_warm(timeout)": (
        "import-time warm-up instance that is discarded without a request"
    ),
}

_DEADLINE_NAME = re.compile(r"(^|_)timeout(_seconds|_secs|_s)?$")
_TIMEOUT_CTORS = frozenset({"ClientTimeout", "Timeout"})
_TIMEOUT_CTOR_FIELDS = frozenset(
    {"total", "connect", "sock_read", "sock_connect", "read", "write", "pool"}
)
# Callee -> index of its positional deadline argument.
_POSITIONAL_DEADLINES = {
    "BlockingCallPolicy": 1,
    "wait_for": 1,
    "ClientTimeout": 0,
    "Timeout": 0,
    "with_timeout": 0,
    "timeout": 0,
}
_LOG_METHODS = frozenset(
    {"debug", "info", "warning", "warn", "error", "exception", "critical", "log"}
)
_MAX_DEPTH = 6
# The repo's own deadline primitives are found by the shape of their signature
# (``enclosing_deadline(seconds)``, ``require_allowance(seconds, ...)``,
# ``run_program(..., timeout)``), never by a hand-kept list of names.
_DEADLINE_PARAM = re.compile(r"(^|_)(timeout|seconds|deadline)(_seconds|_secs|_s)?$")
_RESOLVE_DEPTH = 4


def _is_number(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, int | float) and not isinstance(node.value, bool)
    if isinstance(node, ast.UnaryOp):
        return _is_number(node.operand)
    if isinstance(node, ast.BinOp):
        return _is_number(node.left) and _is_number(node.right)
    return False


def _assignments(nodes) -> dict[str, list[ast.AST]]:
    found: dict[str, list[ast.AST]] = {}
    for node in nodes:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    found.setdefault(target.id, []).append(node.value)
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.value is not None
        ):
            found.setdefault(node.target.id, []).append(node.value)
    return found


# A binding whose value is not followed (a loop target, a nested def, ...).
_OPAQUE = ast.Name(id="<opaque>")
_NEW_SCOPES = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Lambda,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
)


def _scope_nodes(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.AST]:
    """The nodes of a function body's own scope; nested scopes are not entered."""
    found: list[ast.AST] = []
    stack: list[ast.AST] = list(fn.body)
    while stack:
        node = stack.pop()
        found.append(node)
        if not isinstance(node, _NEW_SCOPES):
            stack.extend(ast.iter_child_nodes(node))
    return found


def _local_bindings(
    fn: ast.FunctionDef | ast.AsyncFunctionDef,
) -> dict[str, list[ast.AST]]:
    """Every name the function binds, with the values that can be followed.

    Any other binding (a loop, ``with`` or ``except`` target, tuple unpacking,
    a nested def) shadows the module's name with a value that is not followed.
    ``global``/``nonlocal`` names stay the outer scope's.
    """
    nodes = _scope_nodes(fn)
    found = _assignments(nodes)
    for node in nodes:
        if isinstance(node, ast.NamedExpr):
            found.setdefault(node.target.id, []).append(node.value)
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            increment = ast.BinOp(left=_OPAQUE, op=node.op, right=node.value)
            found.setdefault(node.target.id, []).append(increment)
        stored = (
            [node.id]
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
            else [node.name]
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
            or (isinstance(node, ast.ExceptHandler) and node.name)
            else []
        )
        for name in stored:
            found.setdefault(name, []).append(_OPAQUE)
    shared = {
        name
        for node in nodes
        if isinstance(node, ast.Global | ast.Nonlocal)
        for name in node.names
    }
    return {name: values for name, values in found.items() if name not in shared}


def _is_non_positive(node: ast.AST) -> bool:
    """A literal 0, 0.0 or negative: ``max(x, 0)`` keeps x positive, no more."""
    try:
        value = ast.literal_eval(node)
    except ValueError:
        return False
    return isinstance(value, int | float) and not isinstance(value, bool) and value <= 0


def _dotted(node: ast.AST) -> list[str] | None:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.Attribute):
        head = _dotted(node.value)
        return None if head is None else [*head, node.attr]
    return None


_Param = tuple[str, int | None]


@dataclass(frozen=True)
class _Index:
    """What the whole tree says about names, built once from the parsed files."""

    # module -> module-level name -> the values assigned to it
    constants: dict[str, dict[str, list[ast.AST]]]
    # module -> local alias -> (imported module, imported name or None)
    imports: dict[str, dict[str, tuple[str, str | None]]]
    # (module, function or class name) -> its deadline parameters, each with its
    # call-site position (None if keyword-only); a class means its __init__
    functions: dict[tuple[str, str], frozenset[_Param]]
    # method name -> the deadline parameters of every src method so named
    methods: dict[str, frozenset[_Param]]

    def _origin(self, module: str, node: ast.AST) -> tuple[str, str] | None:
        """The (module, name) a Name or attribute chain names through an import."""
        parts = _dotted(node)
        origin = self.imports.get(module, {}).get(parts[0]) if parts else None
        if parts is None or origin is None:
            return None
        target, attr = origin
        path = [*target.split("."), *([attr] if attr else []), *parts[1:]]
        return ".".join(path[:-1]), path[-1]

    def _defining(
        self, module: str, name: str, defines: Callable[[str, str], bool]
    ) -> tuple[str, str] | None:
        """Follow re-exports of ``module.name`` to the module that defines it."""
        for _ in range(_RESOLVE_DEPTH):
            if defines(module, name):
                return module, name
            origin = self.imports.get(module, {}).get(name)
            if origin is None or origin[1] is None:
                return None
            module, name = origin[0], origin[1]
        return None

    def constant_in(self, module: str, name: str, depth: int = 0) -> bool:
        """Whether ``module.name`` evaluates to a literal, through chains of names."""
        found = self._defining(
            module, name, lambda m, n: n in self.constants.get(m, {})
        )
        if found is None or depth >= _RESOLVE_DEPTH:
            return False
        home, name = found
        names = self.constants[home]
        return any(
            _is_literal_deadline(
                value, names, lambda n: self.resolves_to_number(home, n, depth + 1)
            )
            for value in names[name]
        )

    def resolves_to_number(self, module: str, node: ast.AST, depth: int = 0) -> bool:
        """Whether a Name or attribute chain, via an import, is a literal constant."""
        origin = self._origin(module, node)
        return origin is not None and self.constant_in(*origin, depth)

    def callee_params(self, module: str, func: ast.AST) -> frozenset[_Param]:
        """Deadline parameters of the src callable a call names, by its binding.

        A plain name is this module's own def or an import (alias included),
        and ``mod.f`` an imported module's function. Any other ``obj.m`` has
        a receiver no AST can type, so it matches every src method named ``m``.
        """
        key = (
            (module, func.id)
            if isinstance(func, ast.Name) and (module, func.id) in self.functions
            else self._origin(module, func)
        )
        found = key and self._defining(*key, lambda m, n: (m, n) in self.functions)
        if found:
            return self.functions[found]
        if isinstance(func, ast.Attribute):
            return self.methods.get(func.attr, frozenset())
        return frozenset()


_NO_INDEX = _Index({}, {}, {}, {})


def _is_literal_deadline(
    node: ast.AST | None,
    names: dict[str, list[ast.AST]],
    resolve: Callable[[ast.AST], bool],
    depth: int = 0,
) -> bool:
    """Whether a numeric literal can flow into this expression's value.

    ``resolve`` answers for an imported name or ``module.NAME`` chain.
    """
    if node is None or depth > _MAX_DEPTH:
        return False
    if _is_number(node):
        return True
    nested = depth + 1

    def literal(child: ast.AST) -> bool:
        return _is_literal_deadline(child, names, resolve, nested)

    if isinstance(node, ast.Name):
        if node.id in names:
            return any(literal(v) for v in names[node.id])
        return resolve(node)
    if isinstance(node, ast.Attribute):
        return resolve(node)
    if isinstance(node, ast.BoolOp):
        return any(literal(v) for v in node.values)
    if isinstance(node, ast.IfExp):
        return literal(node.body) or literal(node.orelse)
    if isinstance(node, ast.Tuple):
        return any(literal(v) for v in node.elts)
    if isinstance(node, ast.BinOp):
        # A sum moves a deadline by a fixed amount; any other operation is a
        # scale of whatever it is applied to, fixed only when both sides are.
        if isinstance(node.op, ast.Add | ast.Sub):
            return literal(node.left) or literal(node.right)
        return literal(node.left) and literal(node.right)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id in {"min", "float", "int"}:
            return any(literal(v) for v in node.args)
        if node.func.id == "max":
            # max(x, 0) only keeps x positive; max(x, 30) is a fixed deadline floor
            return any(literal(v) and not _is_non_positive(v) for v in node.args)
        # getattr(obj, "field", 30): the fallback is the value when the field is absent
        return (
            node.func.id == "getattr" and len(node.args) == 3 and literal(node.args[2])
        )
    return False


def _callee(call: ast.Call) -> str:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


@dataclass
class _Scanner(ast.NodeVisitor):
    path: str
    module_names: dict[str, list[ast.AST]]
    module: str = ""
    index: _Index = _NO_INDEX
    findings: set[str] = field(default_factory=set)
    scopes: list[tuple[str, dict[str, list[ast.AST]]]] = field(default_factory=list)
    module_target: str | None = None

    def _literal(self, node: ast.AST | None, names: dict[str, list[ast.AST]]) -> bool:
        return _is_literal_deadline(
            node, names, lambda n: self.index.resolves_to_number(self.module, n)
        )

    def _scope(self) -> str:
        if self.scopes:
            return ".".join(name for name, _ in self.scopes)
        return self.module_target or "<module>"

    def _names(self) -> dict[str, list[ast.AST]]:
        return self.scopes[-1][1] if self.scopes else self.module_names

    def _flag(self, site: str) -> None:
        self.findings.add(f"{self.path}::{self._scope()}::{site}")

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        args = node.args
        positional = [*args.posonlyargs, *args.args]
        defaults = list(
            zip(
                positional[len(positional) - len(args.defaults) :],
                args.defaults,
                strict=True,
            )
        )
        defaults += [
            (arg, default)
            for arg, default in zip(args.kwonlyargs, args.kw_defaults, strict=True)
            if default is not None
        ]
        parameters = {a.arg: [] for a in [*positional, *args.kwonlyargs]}
        local = {**self._names(), **parameters, **_local_bindings(node)}
        self.scopes.append((node.name, local))
        for arg, default in defaults:
            if _DEADLINE_PARAM.search(arg.arg) and self._literal(
                default, self.module_names
            ):
                self._flag(f"def({arg.arg})")
        self.generic_visit(node)
        self.scopes.pop()

    visit_FunctionDef = _visit_function
    visit_AsyncFunctionDef = _visit_function

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scopes.append((node.name, self._names()))
        self.generic_visit(node)
        self.scopes.pop()

    def _visit_module_assignment(self, node: ast.Assign | ast.AnnAssign) -> None:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        names = [t.id for t in targets if isinstance(t, ast.Name)]
        if self.scopes or not names:
            self.generic_visit(node)
            return
        self.module_target = names[0]
        self.generic_visit(node)
        self.module_target = None

    visit_Assign = _visit_module_assignment
    visit_AnnAssign = _visit_module_assignment

    def visit_Call(self, node: ast.Call) -> None:
        callee = _callee(node)
        if not (isinstance(node.func, ast.Attribute) and callee in _LOG_METHODS):
            names = self._names()
            for keyword in node.keywords:
                name = keyword.arg
                if not name:
                    continue
                is_deadline = _DEADLINE_NAME.search(name) or (
                    callee in _TIMEOUT_CTORS and name in _TIMEOUT_CTOR_FIELDS
                )
                if is_deadline and self._literal(keyword.value, names):
                    self._flag(f"{callee}({name})")
            self._visit_repo_deadline_callee(node, callee, names)
            index = _POSITIONAL_DEADLINES.get(callee)
            if (
                index is not None
                and len(node.args) > index
                and self._literal(node.args[index], names)
            ):
                self._flag(f"{callee}[{index}]")
        self.generic_visit(node)

    def _visit_repo_deadline_callee(
        self, node: ast.Call, callee: str, names: dict[str, list[ast.AST]]
    ) -> None:
        """A literal handed to a deadline parameter of a function defined in src/."""
        for param, position in self.index.callee_params(self.module, node.func):
            for keyword in node.keywords:
                if keyword.arg == param and self._literal(keyword.value, names):
                    self._flag(f"{callee}({param})")
            if (
                position is not None
                and len(node.args) > position
                and not any(
                    isinstance(a, ast.Starred) for a in node.args[: position + 1]
                )
                and self._literal(node.args[position], names)
            ):
                self._flag(f"{callee}[{position}]")

    def visit_Dict(self, node: ast.Dict) -> None:
        names = self._names()
        for key, value in zip(node.keys, node.values, strict=True):
            if (
                isinstance(key, ast.Constant)
                and isinstance(key.value, str)
                and _DEADLINE_NAME.search(key.value)
                and self._literal(value, names)
            ):
                self._flag(f"dict[{key.value!r}]")
        self.generic_visit(node)


@dataclass(frozen=True)
class _Parsed:
    path: str
    module: str
    is_package: bool
    tree: ast.Module


def _module_of(path: str) -> tuple[str, bool]:
    parts = path.removesuffix(".py").split("/")
    is_package = parts[-1] == "__init__"
    return ".".join(parts[:-1] if is_package else parts), is_package


@functools.cache
def _parse_tree(root: Path, base: Path) -> tuple[_Parsed, ...]:
    parsed = []
    for file in sorted(root.rglob("*.py")):
        rel = file.relative_to(base).as_posix()
        module, is_package = _module_of(rel)
        tree = ast.parse(file.read_text(encoding="utf-8"))
        parsed.append(_Parsed(rel, module, is_package, tree))
    return tuple(parsed)


def _imports_of(parsed: _Parsed) -> dict[str, tuple[str, str | None]]:
    found: dict[str, tuple[str, str | None]] = {}
    package = parsed.module if parsed.is_package else parsed.module.rpartition(".")[0]
    for node in ast.walk(parsed.tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    found[alias.asname] = (alias.name, None)
                else:
                    head = alias.name.split(".")[0]
                    found[head] = (head, None)
        elif isinstance(node, ast.ImportFrom):
            anchor = ".".join(
                package.split(".")[: len(package.split(".")) - node.level + 1]
            )
            base = anchor if node.level else ""
            target = ".".join(p for p in (base, node.module) if p)
            for alias in node.names:
                if alias.name != "*":
                    found[alias.asname or alias.name] = (target, alias.name)
    return found


def _signature(fn: ast.FunctionDef | ast.AsyncFunctionDef, bound: bool) -> set[_Param]:
    """A def's deadline-shaped parameters, positioned as a call site sees them."""
    positional = [*fn.args.posonlyargs, *fn.args.args][1 if bound else 0 :]
    found: set[_Param] = {
        (arg.arg, position)
        for position, arg in enumerate(positional)
        if _DEADLINE_PARAM.search(arg.arg)
    }
    found |= {
        (a.arg, None) for a in fn.args.kwonlyargs if _DEADLINE_PARAM.search(a.arg)
    }
    return found


def _callables(
    module: str, tree: ast.Module
) -> tuple[dict[tuple[str, str], set[_Param]], dict[str, set[_Param]]]:
    """Every def and class in a module (a class by its __init__), and its methods.

    Functions are keyed by (module, name), nested ones included, so a call is
    matched to the def its name is bound to. A method is keyed by its name only:
    the receiver of ``obj.m()`` cannot be typed from the AST.
    """
    defs = (ast.FunctionDef, ast.AsyncFunctionDef)
    functions: dict[tuple[str, str], set[_Param]] = {}
    methods: dict[str, set[_Param]] = {}
    method_ids: set[int] = set()
    for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
        functions.setdefault((module, cls.name), set())
        for method in (m for m in cls.body if isinstance(m, defs)):
            method_ids.add(id(method))
            static = any(_dotted(d) == ["staticmethod"] for d in method.decorator_list)
            params = _signature(method, bound=not static)
            if method.name == "__init__":
                functions[(module, cls.name)] |= params
            elif params:
                methods.setdefault(method.name, set()).update(params)
    for fn in (n for n in ast.walk(tree) if isinstance(n, defs)):
        if id(fn) not in method_ids:
            key = (module, fn.name)
            functions.setdefault(key, set()).update(_signature(fn, bound=False))
    return functions, methods


def _parse_sources(files: dict[str, str]) -> tuple[_Parsed, ...]:
    parsed = []
    for rel, source in files.items():
        module, is_package = _module_of(rel)
        parsed.append(_Parsed(rel, module, is_package, ast.parse(source)))
    return tuple(parsed)


def _index_of(parsed: tuple[_Parsed, ...]) -> _Index:
    functions: dict[tuple[str, str], set[_Param]] = {}
    methods: dict[str, set[_Param]] = {}
    for item in parsed:
        found_functions, found_methods = _callables(item.module, item.tree)
        functions.update(found_functions)
        for name, params in found_methods.items():
            methods.setdefault(name, set()).update(params)
    return _Index(
        constants={p.module: _assignments(p.tree.body) for p in parsed},
        imports={p.module: _imports_of(p) for p in parsed},
        functions={k: frozenset(v) for k, v in functions.items()},
        methods={k: frozenset(v) for k, v in methods.items()},
    )


@functools.cache
def build_index(root: Path = _SRC, base: Path = _REPO) -> _Index:
    return _index_of(_parse_tree(root, base))


def scan_sources(files: dict[str, str], real_callees: bool = False) -> set[str]:
    """Findings for a small fake tree, indexed from its own files.

    With ``real_callees`` the callables of the real src/ are added, so a
    fixture can import and call ``enclosing_deadline`` without defining it.
    """
    parsed = _parse_sources(files)
    index = _index_of(parsed)
    if real_callees:
        real = build_index()
        index = replace(
            index,
            functions={**real.functions, **index.functions},
            methods={**real.methods, **index.methods},
        )
    return _scan_parsed(parsed, index)


def scan_source(source: str, path: str, index: _Index | None = None) -> set[str]:
    """Findings for one source text; ``index`` defaults to the real src/ tree."""
    tree = ast.parse(source)
    module, _ = _module_of(path)
    scanner = _Scanner(
        path=path,
        module_names=_assignments(tree.body),
        module=module,
        index=build_index() if index is None else index,
    )
    scanner.visit(tree)
    return scanner.findings


def scan_tree(root: Path = _SRC, base: Path = _REPO) -> set[str]:
    return _scan_parsed(_parse_tree(root, base), build_index(root, base))


def _scan_parsed(parsed: tuple[_Parsed, ...], index: _Index) -> set[str]:
    findings: set[str] = set()
    for item in parsed:
        if item.path in CONFIG_OWNERS:
            continue
        scanner = _Scanner(
            path=item.path,
            module_names=_assignments(item.tree.body),
            module=item.module,
            index=index,
        )
        scanner.visit(item.tree)
        findings |= scanner.findings
    return findings


def unexplained(findings: set[str], allowlist: dict[str, str]) -> set[str]:
    return findings - set(allowlist)


def stale(findings: set[str], allowlist: dict[str, str]) -> set[str]:
    return set(allowlist) - findings


_FINDINGS = scan_tree()


# ------------------------------------------------------------- the real tree


def test_no_literal_deadline_outside_the_config_owners():
    leftover = unexplained(_FINDINGS, ALLOWLIST)
    assert not leftover, (
        "A numeric literal reaches an execution-boundary deadline. Read it from a "
        "Settings field (clamped to the caller's enclosing deadline), or add an "
        "ALLOWLIST entry saying why a literal is right here:\n  "
        + "\n  ".join(sorted(leftover))
    )


def test_allowlist_has_no_stale_entry():
    gone = stale(_FINDINGS, ALLOWLIST)
    assert not gone, "ALLOWLIST entries no longer found; remove them:\n  " + (
        "\n  ".join(sorted(gone))
    )


def test_every_allowlist_entry_says_why():
    for key, reason in ALLOWLIST.items():
        assert len(reason.split()) >= 5, key


def test_config_owners_exist():
    for path in CONFIG_OWNERS:
        assert (_REPO / path).is_file(), path


# ----------------------------------------------------- the guard's own proof


@pytest.mark.parametrize(
    "source",
    [
        "async def f(c):\n    await run_with_hard_timeout(c, timeout=30, label='x')",
        "def f():\n    return aiohttp.ClientTimeout(total=10)",
        "def f():\n    return httpx.Timeout(10.0)",
        "P = BlockingCallPolicy('label', 5.0)",
        "async def f(c):\n    await asyncio.wait_for(c, 5)",
        "def f(s):\n    return {'model': 'm', 'timeout': 240}",
        "def f(s):\n    return Client(timeout=s.x or 240)",
        "def f(s, a):\n    return Client(timeout=s.x if a else 240)",
        "def f(s):\n    return Client(timeout=min(s.x, 30))",
        "T = 30\ndef f():\n    return Client(timeout=T)",
        "def f(s):\n    budget = s.total() + 30\n    return run(timeout=budget)",
        "def f(c, timeout_seconds=10):\n    return c",
        "def f(s):\n    return requests.get(s.url, timeout=(3, 10))",
        "def f(p):\n    return p.with_timeout(5.0)",
        "def f(s):\n    return Fetcher(http_timeout_seconds=-1 * -5)",
        "def f(r):\n    return run(timeout=max(0.1, r))",  # a fixed floor
    ],
)
def test_planted_literal_deadline_is_flagged(source):
    assert scan_source(source, "src/planted.py")


@pytest.mark.parametrize(
    "source",
    [
        # A centralised default: a setting, clamped or not.
        "async def f(c):\n    await run_with_hard_timeout("
        "c, timeout=settings_config.tool_execution_timeout_seconds, label='x')",
        "def f(cfg):\n    t = require_allowance(cfg.vendor_timeout_seconds, label='v')\n"
        "    return aiohttp.ClientTimeout(total=t)",
        "from src.llm_runtime.timeouts import apac_client_timeout_seconds\n"
        "def f(s):\n    return {'timeout': apac_client_timeout_seconds(s)}",
        # Not deadlines: a positivity floor, a scale, a log field, None, retries.
        "def f(r):\n    return run(timeout=max(0.0, r))",
        "def f(cfg):\n    return run(timeout=cfg.floor * 1.5)",
        "def f():\n    logger.warning('slow', timeout_seconds=5)",
        "def f():\n    return Client(timeout=None)",
        "MAX_TIMEOUT_RETRIES = 3\ndef f():\n    return Client(max_retries=3)",
    ],
)
def test_centralised_or_non_deadline_value_passes(source):
    assert scan_source(source, "src/planted.py") == set()


def test_finding_key_names_scope_and_parameter():
    source = (
        "class Fetcher:\n"
        "    async def fetch(self, c):\n"
        "        return await run_with_hard_timeout(c, timeout=30, label='x')\n"
    )
    assert scan_source(source, "src/planted.py") == {
        "src/planted.py::Fetcher.fetch::run_with_hard_timeout(timeout)"
    }


def test_a_stale_entry_is_reported_and_a_new_literal_is_not_excused():
    findings = scan_source("def f(c):\n    return run(timeout=30)", "src/planted.py")
    allowlist = {"src/planted.py::g::run(timeout)": "a function that moved away"}
    assert stale(findings, allowlist) == {"src/planted.py::g::run(timeout)"}
    assert unexplained(findings, allowlist) == {"src/planted.py::f::run(timeout)"}


# ------------------------------- names, getattr fallbacks, repo deadline callees

_CFG = "T = 30\nNAME = 'x'\nRATIO: float = 0.5\n"
_USE = "def f():\n    return Client(timeout={arg})\n"
_FLAGGED_AT_F = "src/use.py::f::Client(timeout)"


def _use(header: str, arg: str) -> dict[str, str]:
    return {"src/cfg.py": _CFG, "src/use.py": header + _USE.format(arg=arg)}


@pytest.mark.parametrize(
    ("header", "arg"),
    [
        ("from src.cfg import T\n", "T"),
        ("from src.cfg import T as LIMIT\n", "LIMIT"),
        ("from src import cfg\n", "cfg.T"),
        ("from src import cfg as c\n", "c.T"),
        ("import src.cfg as c\n", "c.T"),
        ("import src.cfg\n", "src.cfg.T"),
        ("from .cfg import T\n", "T"),
        ("from src.cfg import T\n", "T + 5"),
        ("from src.cfg import T\n", "T or 1"),
        ("from src.cfg import RATIO\n", "RATIO"),
        ("from src.cfg import T\n", "max(T, 0.1)"),  # a fixed floor
        ("from src.cfg import T\n", "T * 1.5"),  # constant times constant
    ],
)
def test_an_imported_constant_is_followed_to_its_literal(header, arg):
    assert scan_sources(_use(header, arg)) == {_FLAGGED_AT_F}


def test_a_config_owners_constant_is_not_exempt_once_imported_elsewhere():
    files = {
        "src/config.py": "TIMEOUT = 45\n",
        "src/use.py": "from src.config import TIMEOUT\n" + _USE.format(arg="TIMEOUT"),
    }
    assert "src/config.py" in CONFIG_OWNERS
    assert scan_sources(files) == {_FLAGGED_AT_F}


def test_a_constant_re_exported_through_a_package_is_followed():
    files = {
        "src/pkg/__init__.py": "from src.pkg.defaults import T\n",
        "src/pkg/defaults.py": "T = 9\n",
        "src/use.py": "from src.pkg import T\n" + _USE.format(arg="T"),
    }
    assert scan_sources(files) == {_FLAGGED_AT_F}


def test_a_relative_import_inside_a_package_is_resolved():
    files = {
        "src/pkg/__init__.py": "",
        "src/pkg/defaults.py": "T = 9\n",
        "src/pkg/use.py": "from . import defaults\n" + _USE.format(arg="defaults.T"),
    }
    assert scan_sources(files) == {"src/pkg/use.py::f::Client(timeout)"}


@pytest.mark.parametrize(
    ("header", "arg"),
    [
        ("from src.cfg import NAME\n", "NAME"),  # not numeric
        ("from src.cfg import MISSING\n", "MISSING"),  # not defined anywhere
        ("from other.lib import T\n", "T"),  # outside the tree
        ("from src.cfg import T\n", "max(s.x, 0)"),  # a positivity floor
        ("from src.cfg import RATIO\n", "s.x * RATIO"),  # a fraction of a setting
        ("from src import cfg\n", "cfg.NAME"),
        ("from src import cfg\n", "cfg.T.real"),  # not the constant itself
        ("import src.cfg\n", "src.cfg"),  # the module, not a member
        ("from src.cfg import T\n", "settings.T"),  # a different object
    ],
)
def test_an_unresolvable_or_non_numeric_import_passes(header, arg):
    assert scan_sources(_use(header, arg)) == set()


def test_a_parameter_shadowing_an_imported_constant_is_not_the_constant():
    files = {
        "src/cfg.py": _CFG,
        "src/use.py": "from src.cfg import T\n"
        "def f(T):\n    return Client(timeout=T)\n",
    }
    assert scan_sources(files) == set()


@pytest.mark.parametrize(
    "arg",
    [
        "getattr(s, 'x', 30)",
        "getattr(s, 'x', 30.0)",
        "getattr(s, 'x', -1 * -30)",
        "getattr(s, 'x', 30) + 5",
        "getattr(s, 'x', 30) or s.y",
        "min(getattr(s, 'x', 30), s.cap)",
    ],
)
def test_a_getattr_numeric_fallback_counts_as_a_literal(arg):
    source = f"def f(s):\n    return Client(timeout={arg})\n"
    assert scan_source(source, "src/use.py") == {_FLAGGED_AT_F}


def test_a_getattr_fallback_through_a_local_or_an_imported_constant_counts():
    files = {
        "src/cfg.py": _CFG,
        "src/use.py": "from src.cfg import T\n"
        "def f(s):\n    return Client(timeout=getattr(s, 'x', T))\n",
    }
    assert scan_sources(files) == {_FLAGGED_AT_F}


@pytest.mark.parametrize(
    "arg",
    [
        "getattr(s, 'x')",  # two arguments: raises, never falls back
        "getattr(s, 'x', None)",
        "getattr(s, 'x', s.y)",
        "getattr(s, 'x', 'ten')",
        "getattr(s, 'x', False)",
        "getattr(s, 'x', default=30)",  # getattr takes no keywords
    ],
)
def test_a_getattr_without_a_numeric_fallback_passes(arg):
    source = f"def f(s):\n    return Client(timeout={arg})\n"
    assert scan_source(source, "src/use.py") == set()


# The real deadline primitives, imported as src/ code imports them.
_REAL_IMPORTS = (
    "from src.llm_runtime.timeouts import (\n"
    "    clamp_to_operational_deadline, enclosing_deadline,\n"
    "    require_allowance, tool_call_deadline,\n"
    ")\n"
    "from src.subprocesses import run_program\n"
)


@pytest.mark.parametrize(
    ("source", "key"),
    [
        ("enclosing_deadline(240)", "enclosing_deadline[0]"),
        ("enclosing_deadline(seconds=240)", "enclosing_deadline(seconds)"),
        ("require_allowance(30, label='x')", "require_allowance[0]"),
        (
            "require_allowance(default_seconds=30, label='x')",
            "require_allowance(default_seconds)",
        ),
        ("clamp_to_operational_deadline(12.5)", "clamp_to_operational_deadline[0]"),
        ("tool_call_deadline(90)", "tool_call_deadline[0]"),
        ("run_program(['git'], timeout=5)", "run_program(timeout)"),
        ("s.check_within(seconds=5)", "check_within(seconds)"),  # a method
        ("enclosing_deadline(getattr(s, 'x', 240))", "enclosing_deadline[0]"),
        ("enclosing_deadline(min(s.x, 240))", "enclosing_deadline[0]"),
    ],
)
def test_a_literal_handed_to_a_repo_deadline_callee_is_flagged(source, key):
    files = {"src/use.py": f"{_REAL_IMPORTS}def f(s):\n    return {source}\n"}
    assert scan_sources(files, real_callees=True) == {f"src/use.py::f::{key}"}


def test_a_repo_deadline_callee_is_followed_through_a_constant_and_a_method():
    files = {
        "src/cfg.py": _CFG,
        "src/lib.py": "class Gate:\n"
        "    def __init__(self, hard_deadline): ...\n"
        "    def hold(self, label, wait_seconds): ...\n"
        "    @staticmethod\n"
        "    def peek(timeout_s): ...\n",
        "src/use.py": "from src.cfg import T\n"
        "from src.lib import Gate\n"
        "def f(g):\n    g.hold('x', T)\n    Gate(30)\n    Gate.peek(2)\n",
    }
    assert scan_sources(files) == {
        "src/use.py::f::hold[1]",
        "src/use.py::f::Gate[0]",
        "src/use.py::f::peek[0]",
    }


@pytest.mark.parametrize(
    "source",
    [
        "enclosing_deadline(seconds)",  # a parameter passed through
        "enclosing_deadline(seconds=seconds)",
        "enclosing_deadline(clamp_to_operational_deadline(seconds) * 0.5)",
        "enclosing_deadline(max(seconds, 0.0))",
        "enclosing_deadline(max(seconds, min_useful_seconds(s.cap)))",
        "enclosing_deadline(s.total_seconds - clamp(seconds))",
        "require_allowance(apac_client_timeout_seconds(s), label='x')",
        "require_allowance(s.vendor_timeout_seconds, label='x')",
        "enclosing_deadline(getattr(s, 'x', None))",
        "enclosing_deadline(*args)",
        "run_program(['git'], timeout=None)",
        "something_else(30)",  # no deadline parameter anywhere in src/
    ],
)
def test_a_deadline_callee_given_a_settings_value_or_pass_through_passes(source):
    files = {
        "src/use.py": f"{_REAL_IMPORTS}def f(s, seconds, args):\n    return {source}\n"
    }
    assert scan_sources(files, real_callees=True) == set()


def test_the_repo_deadline_callees_are_derived_from_src_not_listed():
    index = build_index()
    timeouts = "src.llm_runtime.timeouts"
    for name in (
        "enclosing_deadline",
        "require_allowance",
        "clamp_to_operational_deadline",
        "tool_call_deadline",
    ):
        assert index.functions[(timeouts, name)], name
    assert index.functions[(timeouts, "enclosing_deadline")] == {("seconds", 0)}
    assert ("timeout", None) in index.functions[("src.subprocesses", "run_program")]
    assert ("seconds", None) in index.methods["check_within"]  # keyword-only


def test_a_log_call_is_still_not_a_deadline_boundary():
    files = {
        "src/use.py": "def f(s):\n    logger.info('x', seconds=5, timeout_seconds=5)\n"
    }
    assert scan_sources(files, real_callees=True) == set()


def test_scan_tree_resolves_imports_on_a_real_directory(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "cfg.py").write_text("T = 30\n", encoding="utf-8")
    (tmp_path / "pkg" / "use.py").write_text(
        "from pkg.cfg import T\ndef f():\n    return Client(timeout=T)\n",
        encoding="utf-8",
    )
    assert scan_tree(tmp_path, tmp_path) == {"pkg/use.py::f::Client(timeout)"}


# ------------------ aliases, constant chains, max(), defaults and lexical scope

_LIB = (
    "def enclosing_deadline(seconds): ...\n"
    "def require_allowance(default_seconds, *, label): ...\n"
    "class Gate:\n"
    "    def hold(self, label, wait_seconds): ...\n"
)


def _with_lib(use: str, **extra: str) -> dict[str, str]:
    return {"src/lib.py": _LIB, **extra, "src/use.py": use}


@pytest.mark.parametrize(
    ("use", "key"),
    [
        (
            "from src.lib import enclosing_deadline as ed\ndef f():\n    ed(30)\n",
            "ed[0]",
        ),
        (
            "from src.lib import require_allowance as ra\n"
            "def f():\n    ra(default_seconds=30, label='x')\n",
            "ra(default_seconds)",
        ),
        (
            "import src.lib as L\ndef f():\n    L.enclosing_deadline(30)\n",
            "enclosing_deadline[0]",
        ),
    ],
)
def test_an_aliased_deadline_callee_is_resolved_through_its_import(use, key):
    assert scan_sources(_with_lib(use)) == {f"src/use.py::f::{key}"}


@pytest.mark.parametrize(
    "cfg",
    [
        "A = 30\nB = A\n",  # chained in the defining module
        "A = 30\nB = A + 5\n",
        "A = 30\nB = min(A, 60)\n",
        "A = 30\nB: float = float(A)\n",
        "A = 30\nB = A * 0.5\n",  # constant times constant is still constant
    ],
)
def test_a_constant_chain_is_followed_across_an_import(cfg):
    files = {
        "src/cfg.py": cfg,
        "src/use.py": "from src.cfg import B\n" + _USE.format(arg="B"),
    }
    assert scan_sources(files) == {_FLAGGED_AT_F}


def test_a_constant_chain_through_a_second_module_is_followed():
    files = {
        "src/base.py": "A = 30\n",
        "src/cfg.py": "from src.base import A\nB = A\n",
        "src/use.py": "from src.cfg import B\n"
        "from src.lib import enclosing_deadline\n"
        "def f():\n    enclosing_deadline(seconds=B)\n",
        "src/lib.py": _LIB,
    }
    assert scan_sources(files) == {"src/use.py::f::enclosing_deadline(seconds)"}


@pytest.mark.parametrize(
    "cfg",
    [
        "A = 'x'\nB = A\n",
        "A = settings.x\nB = A\n",
        "A = settings.x\nB = A * 0.5\n",  # a fraction of a setting
        "B = B\n",  # self-reference terminates
    ],
)
def test_a_constant_chain_without_a_literal_sum_passes(cfg):
    files = {
        "src/cfg.py": cfg,
        "src/use.py": "from src.cfg import B\n" + _USE.format(arg="B"),
    }
    assert scan_sources(files) == set()


def test_a_cyclic_re_export_terminates():
    files = {
        "src/a.py": "from src.b import X\n",
        "src/b.py": "from src.a import X\n",
        "src/use.py": "from src.a import X\n" + _USE.format(arg="X"),
    }
    assert scan_sources(files) == set()


@pytest.mark.parametrize(
    ("arg", "flagged"),
    [
        ("max(30, x)", True),
        ("max(x, 30)", True),
        ("max(x, 0.1)", True),  # a positive floor extends a spent deadline
        ("max(x, T)", True),  # through an imported constant
        ("max(x, getattr(s, 'y', 5))", True),
        ("max(0, x)", False),  # a positivity floor
        ("max(x, 0.0)", False),
        ("max(x, -1)", False),
        ("max(x, min_useful_seconds(s))", False),  # a computed lower bound
        ("max(x, y)", False),
        ("min(x, s.cap) * FRACTION", False),
        ("s.cap * FRACTION", False),
        ("deadline_margin(s.cap)", False),
    ],
)
def test_max_flags_a_fixed_value_but_not_a_positivity_floor(arg, flagged):
    files = {
        "src/cfg.py": _CFG,
        "src/use.py": "from src.cfg import T\n"
        "from src.lib import require_allowance\n"
        f"def f(s, x, y):\n    require_allowance({arg}, label='x')\n",
        "src/lib.py": _LIB,
    }
    expected = {"src/use.py::f::require_allowance[0]"} if flagged else set()
    assert scan_sources(files) == expected


@pytest.mark.parametrize(
    ("signature", "site"),
    [
        ("seconds=30", "def(seconds)"),
        ("*, deadline=30", "def(deadline)"),
        ("wait_seconds=30.0", "def(wait_seconds)"),
        ("hard_deadline=T", "def(hard_deadline)"),
        ("timeout_s=min(T, 9)", "def(timeout_s)"),
    ],
)
def test_a_literal_default_for_a_deadline_shaped_parameter_is_flagged(signature, site):
    files = {
        "src/cfg.py": _CFG,
        "src/use.py": f"from src.cfg import T\ndef f({signature}): ...\n",
    }
    assert scan_sources(files) == {f"src/use.py::f::{site}"}


@pytest.mark.parametrize(
    "signature",
    ["seconds=None", "seconds=DEFAULT", "retries=3", "seconds_left=30", "label='30'"],
)
def test_a_non_literal_default_or_a_non_deadline_parameter_passes(signature):
    files = {"src/use.py": f"def f({signature}): ...\n"}
    assert scan_sources(files) == set()


@pytest.mark.parametrize(
    "body",
    [
        # A nested def's local is not the outer function's.
        "    T = s.x\n    def g():\n        T = 30\n        return T\n"
        "    Client(timeout=T)\n",
        # Loop, with, except and tuple targets shadow the module constant.
        "    for T in s.items:\n        Client(timeout=T)\n",
        "    with s.cm as T:\n        Client(timeout=T)\n",
        "    try:\n        pass\n    except E as T:\n        Client(timeout=T)\n",
        "    _, T = s.pair\n    Client(timeout=T)\n",
        "    (T := s.x)\n    Client(timeout=T)\n",
        "    def T(): ...\n    Client(timeout=T)\n",
        # A comprehension's variable stays inside it.
        "    _ = [T for T in s.items]\n    Client(timeout=s.x)\n",
    ],
)
def test_a_local_binding_shadows_the_module_constant(body):
    source = f"T = 30\ndef f(s):\n{body}"
    assert scan_source(source, "src/use.py", _NO_INDEX) == set()


@pytest.mark.parametrize(
    "body",
    [
        "    Client(timeout=T)\n",  # the module constant itself
        "    def g():\n        Client(timeout=T)\n    g()\n",  # a closure
        "    global T\n    T = s.x\n    Client(timeout=T)\n",  # global is shared
        "    t = s.x\n    t += 30\n    Client(timeout=t)\n",  # a fixed increment
        "    t = (u := 30)\n    Client(timeout=u)\n",
    ],
)
def test_a_literal_reaching_through_scope_is_still_flagged(body):
    source = f"T = 30\ndef f(s):\n{body}"
    assert scan_source(source, "src/use.py", _NO_INDEX)


def test_a_free_function_is_not_mistaken_for_a_same_named_method():
    use = "def hold(label, n): ...\ndef f():\n    hold('x', 30)\n"
    assert scan_sources(_with_lib(use)) == set()


def test_an_unimported_free_function_on_a_receiver_is_not_a_deadline_callee():
    use = "def f(other):\n    other.enclosing_deadline(30)\n"
    assert scan_sources(_with_lib(use)) == set()


def test_a_method_on_an_unknown_receiver_is_matched_by_name():
    # Residual: without types the receiver is unknown, so any method named like
    # a src method with a deadline parameter is treated as that method.
    use = "def f(other):\n    other.hold('x', 30)\n"
    assert scan_sources(_with_lib(use)) == {"src/use.py::f::hold[1]"}
