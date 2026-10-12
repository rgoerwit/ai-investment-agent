"""Tool-call reason codes have one owner, and every other module imports them.

The codes a failure observer records and the model is told were spelled as
bare literals in the runtime, the graph tool node, the junior relay and the
official-document tool, and one fact (a fail-closed inspection backend error)
had two spellings. A literal that drifts from the owner's spelling silently
splits one outcome into two in the ledger. This scans ``src`` by AST: no string
constant other than a docstring may be a code, bare or behind a ``TOOL_ERROR:``
or ``TOOL_BLOCKED:`` prefix, outside the owner; retired spellings appear nowhere.
"""

from __future__ import annotations

import ast
from functools import cache
from pathlib import Path

import pytest

import src.tooling.runtime as tool_runtime

_SRC = Path(__file__).resolve().parents[2] / "src"
_OWNER = _SRC / "tooling" / "runtime.py"


def _owner_codes() -> tuple[str, ...]:
    """The owner's codes, read from it: each module-level ``NAME = "NAME"``."""
    tree = ast.parse(_OWNER.read_text(encoding="utf-8"))
    return tuple(
        node.targets[0].id
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Constant)
        and node.value.value == node.targets[0].id
    )


CODES = _owner_codes()
# Every code a tool, hook or the classifier raises with: the scan below is
# only as wide as what the owner defines, so a missing one is caught here.
_RAISED = (
    "TOOL_TIMEOUT",
    "TOOL_EXECUTION_FAILED",
    "INSPECTION_TIMEOUT",
    "INSPECTION_FAILED",
    "INSPECTION_BLOCKED",
)
RETIRED = ("INSPECTION_BACKEND_ERROR", "OFFICIAL_DOCUMENT_BLOCKED")
_PREFIXES = ("TOOL_ERROR:", "TOOL_BLOCKED:")


def _docstring_nodes(tree: ast.AST) -> set[int]:
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(
            node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
        ):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
            ):
                ids.add(id(body[0].value))
    return ids


def _code_spelled_by(value: str) -> str:
    text = value.strip()
    for prefix in _PREFIXES:
        if text.startswith(prefix):
            text = text.removeprefix(prefix).strip()
    return text


@cache
def _string_constants(path: Path) -> tuple[tuple[int, str], ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    docstrings = _docstring_nodes(tree)
    return tuple(
        (node.lineno, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    )


_SOURCES = sorted(_SRC.rglob("*.py"))


@pytest.mark.parametrize("code", _RAISED)
def test_the_owner_defines_each_raised_code_under_its_own_name(code):
    assert code in CODES
    assert getattr(tool_runtime, code) == code


@pytest.mark.parametrize("code", CODES)
def test_no_module_outside_the_owner_spells_the_code(code):
    offenders = [
        f"{path.relative_to(_SRC.parent)}:{line}"
        for path in _SOURCES
        if path != _OWNER
        for line, value in _string_constants(path)
        if _code_spelled_by(value) == code
    ]
    assert offenders == [], f"import {code} from src.tooling.runtime"


@pytest.mark.parametrize("code", RETIRED)
def test_a_retired_spelling_is_gone(code):
    offenders = [
        f"{path.relative_to(_SRC.parent)}:{line}"
        for path in _SOURCES
        for line, value in _string_constants(path)
        if code in value
    ]
    assert offenders == []


def test_the_scan_sees_a_planted_literal(tmp_path):
    """The scanner itself bites: a prefixed literal is read as its code."""
    planted = tmp_path / "planted.py"
    planted.write_text(
        '"""INSPECTION_FAILED in a docstring is prose."""\n'
        'X = "TOOL_ERROR: TOOL_TIMEOUT"\n',
        encoding="utf-8",
    )
    spelled = [_code_spelled_by(value) for _, value in _string_constants(planted)]
    assert spelled == ["TOOL_TIMEOUT"]
