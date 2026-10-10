"""Operational code reads one deadline; only LLM admission reads the LLM one.

There used to be two public clamps, one per ContextVar, and each call site
picked one: MCP and the official-document tool read the tool runner's
deadline, while Tavily, market data, FX and every other operational call read
only the LLM enclosing deadline and were cut off by the runner as a bare
``TOOL_TIMEOUT``. Now ``src/llm_runtime/timeouts.py`` exposes one operational
reader (``operational_deadline`` and the clamps built on it), and this scan
keeps a new call site from reaching past it:

- ``current_enclosing_deadline`` (the LLM-only reader) is used only where an
  LLM request is admitted; the injection judge must never see a tool deadline.
- The two ContextVars are touched only by their module.
- Only ``ToolExecutionService`` binds the tool-call deadline.
- The retired clamp names do not come back.

AST, not text: comments that explain the retired pattern must not trip it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"

# Each entry is an LLM admission or LLM workflow boundary, with the reason.
LLM_DEADLINE_READERS = {
    # Defines it.
    "llm_runtime/timeouts.py",
    # Sizes each LLM request from the invocation's (or node's) deadline.
    "llm_runtime/dispatch.py",
    # The runtime invocation loop: retries admitted against the node deadline.
    "agents/runtime.py",
    # The article workflow's own LLM deadline (refusal and step bound).
    "article_writer.py",
}
CONTEXTVAR_OWNER = "llm_runtime/timeouts.py"
TOOL_DEADLINE_BINDERS = {CONTEXTVAR_OWNER, "tooling/runtime.py"}
RETIRED = {
    "clamp_to_enclosing_deadline",
    "clamp_to_tool_call_deadline",
    "current_tool_call_deadline",
}


def _names(tree: ast.AST) -> list[tuple[str, int]]:
    """Every identifier used, imported or called (Name, Attribute, alias)."""
    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            found.append((node.id, node.lineno))
        elif isinstance(node, ast.Attribute):
            found.append((node.attr, node.lineno))
        elif isinstance(node, ast.alias):
            found.append((node.name.rsplit(".", 1)[-1], getattr(node, "lineno", 0)))
    return found


def _offenders(name: str, allowed: set[str], root: Path = SRC) -> list[str]:
    offenders = []
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        if rel in allowed:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        offenders.extend(
            f"{rel}:{line}" for found, line in _names(tree) if found == name
        )
    return offenders


def test_only_llm_admission_reads_the_llm_enclosing_deadline():
    assert _offenders("current_enclosing_deadline", LLM_DEADLINE_READERS) == []


@pytest.mark.parametrize("var", ["_ENCLOSING_DEADLINE", "_TOOL_CALL_DEADLINE"])
def test_deadline_contextvars_stay_private_to_their_module(var):
    assert _offenders(var, {CONTEXTVAR_OWNER}) == []


def test_only_the_tool_service_binds_the_tool_call_deadline():
    assert _offenders("tool_call_deadline", TOOL_DEADLINE_BINDERS) == []


@pytest.mark.parametrize("name", sorted(RETIRED))
def test_retired_clamps_stay_retired(name):
    assert _offenders(name, set()) == []


# --- The guard bites ----------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "name", "allowed"),
    [
        (
            "from src.llm_runtime.timeouts import current_enclosing_deadline\n"
            "def size():\n    return current_enclosing_deadline()\n",
            "current_enclosing_deadline",
            LLM_DEADLINE_READERS,
        ),
        (
            "import src.llm_runtime.timeouts as t\n"
            "def size():\n    return t._TOOL_CALL_DEADLINE.get()\n",
            "_TOOL_CALL_DEADLINE",
            {CONTEXTVAR_OWNER},
        ),
        (
            "from src.llm_runtime import timeouts\n"
            "def run():\n    with timeouts.tool_call_deadline(5):\n        pass\n",
            "tool_call_deadline",
            TOOL_DEADLINE_BINDERS,
        ),
        (
            "from src.llm_runtime.timeouts import clamp_to_enclosing_deadline as c\n",
            "clamp_to_enclosing_deadline",
            set(),
        ),
    ],
)
def test_the_scan_catches_a_planted_offender(tmp_path, source, name, allowed):
    (tmp_path / "planted.py").write_text(source, encoding="utf-8")
    assert _offenders(name, allowed, root=tmp_path) != []


def test_a_comment_naming_a_retired_clamp_is_not_an_offender(tmp_path):
    (tmp_path / "doc.py").write_text(
        "# clamp_to_enclosing_deadline was retired in October 2026\nx = 1\n",
        encoding="utf-8",
    )
    assert _offenders("clamp_to_enclosing_deadline", set(), root=tmp_path) == []
