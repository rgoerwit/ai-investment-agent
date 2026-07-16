from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from src.data_block_utils import (
    BLOCK_SHAPES,
    BlockShape,
    fenced_end,
    fenced_start,
)
from src.eval.prompt_contracts import PROMPT_CONTRACTS, Shape, prompt_text

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_BLOCK_NAME_PATTERN = "|".join(re.escape(name) for name in sorted(BLOCK_SHAPES))
_FENCED_MARKER_LITERAL_RE = re.compile(
    rf"-{{2,}}\s*(?:START|END)\s+(?:{_BLOCK_NAME_PATTERN})\b"
)
_MARKER_LITERAL_ALLOWLIST = {
    "src/data_block_utils.py",
    "src/report_generator.py",
    "src/agents/forensic_repair.py",
}

_PROMPT_FOR_FENCED_BLOCK = {
    "DATA_BLOCK": "fundamentals_analyst",
    "PM_BLOCK": "portfolio_manager",
    "VALUE_TRAP_BLOCK": "value_trap_detector",
    "VALUATION_PARAMS": "valuation_calculator",
    "VALUATION_SCENARIOS": "valuation_calculator",
    "KILL_CRITERIA": "bear_researcher",
}


def _docstring_value_nodes(tree: ast.AST) -> set[ast.Constant]:
    nodes: set[ast.Constant] = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list) or not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            nodes.add(first.value)
    return nodes


def _string_literals(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    docstring_nodes = _docstring_value_nodes(tree)
    values: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node in docstring_nodes:
                continue
            values.append(node.value)
    return values


@pytest.mark.parametrize(
    ("block_name", "prompt_key"),
    sorted(_PROMPT_FOR_FENCED_BLOCK.items()),
)
def test_fenced_prompts_emit_canonical_markers(
    block_name: str,
    prompt_key: str,
) -> None:
    msg = prompt_text(prompt_key)
    assert fenced_start(block_name) in msg
    assert fenced_end(block_name) in msg


def test_prompt_contract_shapes_match_block_registry() -> None:
    for contract in PROMPT_CONTRACTS:
        if contract.block_name is None:
            continue
        registry_shape = BLOCK_SHAPES[contract.block_name]
        expected = (
            Shape.FENCED_BLOCK
            if registry_shape is BlockShape.FENCED
            else Shape.UNFENCED_BLOCK
        )
        assert contract.shape is expected


def test_auditor_prompt_does_not_teach_inline_forensic_shorthand() -> None:
    assert "FORENSIC_DATA_BLOCK: STATUS=" not in prompt_text("global_forensic_auditor")


def test_src_fenced_marker_literals_stay_centralized() -> None:
    offenders: list[str] = []
    for path in sorted((_PROJECT_ROOT / "src").rglob("*.py")):
        relative = path.relative_to(_PROJECT_ROOT).as_posix()
        if relative in _MARKER_LITERAL_ALLOWLIST:
            continue
        for literal in _string_literals(path):
            if _FENCED_MARKER_LITERAL_RE.search(literal):
                offenders.append(relative)
                break

    assert offenders == []
