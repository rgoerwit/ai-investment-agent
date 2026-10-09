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
local variable, ``x or 240``, ``a if c else 240``, ``min(x, 30)`` and ``x + 30``.
``max(x, 0.1)`` is a positivity floor and ``x * 1.5`` a scale, so neither counts.

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
import re
from dataclasses import dataclass, field
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
    "src/llm_runtime/timeouts.py": "the LLM budget policy and its named ceilings",
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
    "src/agents/consultant_nodes.py::create_consultant_node.consultant_node::run_with_hard_timeout(timeout)": (
        "the settings-derived consultant total plus a fixed 30 s grace "
        "(CONSULTANT_HARD_TIMEOUT_GRACE_SECONDS): the gap that lets the "
        "cooperative deadline win over the hard node bound, a relation not a knob"
    ),
    "src/llm_runtime/construction.py::build_legacy_model::create_gemini_model(timeout)": (
        "fallback mirrors the Health Check seat's client_timeout_seconds "
        "(10.0, src/llm_runtime/seats.py), which always sets it; legacy path only"
    ),
    # Not on the analysis path, or not a deadline at all.
    "src/eval/git_meta.py::_git_output::run(timeout)": (
        "local git subprocess for eval provenance metadata: no network, not on "
        "the analysis path; a setting would be dead tuning surface"
    ),
    "src/ibkr/analysis_index.py::load_latest_analyses._heartbeat_worker::wait(timeout)": (
        "Event.wait used as a 30 s heartbeat interval (a polling cadence, "
        "out of scope), not a deadline on any work"
    ),
    "src/tools/shared.py::<module>::_DDGS_warm(timeout)": (
        "import-time warm-up instance that is discarded without a request"
    ),
    "src/tools/shared.py::fetch_with_timeout::def(timeout_seconds)": (
        "a helper with no caller in src/, tests/ or scripts/; a default "
        "nothing exercises (deletion is the better fix, left to a later change)"
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


def _is_literal_deadline(
    node: ast.AST | None, names: dict[str, list[ast.AST]], depth: int = 0
) -> bool:
    """Whether a numeric literal can flow into this expression's value."""
    if node is None or depth > _MAX_DEPTH:
        return False
    if _is_number(node):
        return True
    nested = depth + 1
    if isinstance(node, ast.Name):
        return any(
            _is_literal_deadline(v, names, nested) for v in names.get(node.id, ())
        )
    if isinstance(node, ast.BoolOp):
        return any(_is_literal_deadline(v, names, nested) for v in node.values)
    if isinstance(node, ast.IfExp):
        return _is_literal_deadline(node.body, names, nested) or _is_literal_deadline(
            node.orelse, names, nested
        )
    if isinstance(node, ast.Tuple):
        return any(_is_literal_deadline(v, names, nested) for v in node.elts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add | ast.Sub):
        return _is_literal_deadline(node.left, names, nested) or _is_literal_deadline(
            node.right, names, nested
        )
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"min", "float", "int"}
    ):
        return any(_is_literal_deadline(v, names, nested) for v in node.args)
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
    findings: set[str] = field(default_factory=set)
    scopes: list[tuple[str, dict[str, list[ast.AST]]]] = field(default_factory=list)
    module_target: str | None = None

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
        local = {**self._names(), **_assignments(ast.walk(node))}
        self.scopes.append((node.name, local))
        for arg, default in defaults:
            if _DEADLINE_NAME.search(arg.arg) and _is_literal_deadline(
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
                if is_deadline and _is_literal_deadline(keyword.value, names):
                    self._flag(f"{callee}({name})")
            index = _POSITIONAL_DEADLINES.get(callee)
            if (
                index is not None
                and len(node.args) > index
                and _is_literal_deadline(node.args[index], names)
            ):
                self._flag(f"{callee}[{index}]")
        self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict) -> None:
        names = self._names()
        for key, value in zip(node.keys, node.values, strict=True):
            if (
                isinstance(key, ast.Constant)
                and isinstance(key.value, str)
                and _DEADLINE_NAME.search(key.value)
                and _is_literal_deadline(value, names)
            ):
                self._flag(f"dict[{key.value!r}]")
        self.generic_visit(node)


def scan_source(source: str, path: str) -> set[str]:
    tree = ast.parse(source)
    scanner = _Scanner(path=path, module_names=_assignments(tree.body))
    scanner.visit(tree)
    return scanner.findings


def scan_tree(root: Path = _SRC) -> set[str]:
    findings: set[str] = set()
    for file in sorted(root.rglob("*.py")):
        rel = file.relative_to(_REPO).as_posix()
        if rel in CONFIG_OWNERS:
            continue
        findings |= scan_source(file.read_text(encoding="utf-8"), rel)
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
        "from src.llm_runtime.timeouts import APAC_CLIENT_TIMEOUT_SECONDS\n"
        "def f():\n    return {'timeout': APAC_CLIENT_TIMEOUT_SECONDS}",
        # Not deadlines: a positivity floor, a scale, a log field, None, retries.
        "def f(r):\n    return run(timeout=max(0.1, r))",
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
