"""Mermaid blocks in the docs must keep their quotable strings quoted.

Unquoted labels *happen* to parse while they contain nothing special, which is
exactly what makes them a trap: the README's one unquoted edge label parsed fine
until someone added a parenthesis, at which point the whole diagram failed to
render with a bare "Parse error on line N". Verified against mermaid 11:

    |Independent Forensic Report|              -> parses
    |Independent Forensic Report (bounded)|    -> Parse error
    |"Independent Forensic Report (bounded)"|  -> parses

A GitHub-rendered diagram has no test that runs mermaid itself (that needs node +
a headless browser), so this is a structural lint: quote everything quotable and
the class of failure cannot occur.
"""

import re
from pathlib import Path

import pytest

_DOC_ROOT = Path(__file__).resolve().parents[2]
_MERMAID_BLOCK = re.compile(r"```mermaid\n(.*?)```", re.S)
_SUBGRAPH = re.compile(r"^\s*subgraph\s+(?P<title>.+?)\s*$", re.M)
_EDGE_LABEL = re.compile(r"\|([^|\n]*)\|")
_NODE_START = re.compile(r"([A-Za-z_]\w*)\s*([\[{(]{1,2})")
_CLOSERS = "]})"


def _markdown_files() -> list[Path]:
    files = [_DOC_ROOT / "README.md"]
    files += sorted((_DOC_ROOT / "docs").glob("*.md"))
    security = _DOC_ROOT / "SECURITY.md"
    if security.exists():
        files.append(security)
    return [path for path in files if path.exists()]


def _mermaid_blocks() -> list[tuple[Path, str]]:
    return [
        (path, block)
        for path in _markdown_files()
        for block in _MERMAID_BLOCK.findall(path.read_text(encoding="utf-8"))
    ]


def _node_labels(block: str) -> list[str]:
    """Extract node labels, scanning to the matching close bracket.

    A non-greedy regex stops at the first ``)`` *inside* a quoted label
    (``"Market Analyst<br/>(Technical)"``) and reports a false positive, so the
    shape body is matched by depth-counting with quote awareness instead.
    """
    labels: list[str] = []
    index = 0
    while (match := _NODE_START.search(block, index)) is not None:
        cursor = match.end()
        start = cursor
        depth = 1
        in_string = False
        while cursor < len(block) and depth:
            char = block[cursor]
            if char == '"':
                in_string = not in_string
            elif not in_string and char in "[{(":
                depth += 1
            elif not in_string and char in _CLOSERS:
                depth -= 1
            cursor += 1
        labels.append(block[start : cursor - 1].strip())
        index = cursor
    return labels


def _is_quoted(text: str) -> bool:
    text = text.strip()
    return len(text) >= 2 and text.startswith('"') and text.endswith('"')


def test_documentation_contains_mermaid_to_check() -> None:
    """Guard the guard: a silently empty corpus would pass everything below."""
    assert _mermaid_blocks(), "no mermaid blocks found — did the extractor break?"


@pytest.mark.parametrize(
    ("path", "block"), _mermaid_blocks(), ids=lambda value: getattr(value, "name", "")
)
def test_every_quotable_string_is_quoted(path: Path, block: str) -> None:
    unquoted: list[str] = []
    unquoted += [
        f"node label: {label}" for label in _node_labels(block) if not _is_quoted(label)
    ]
    unquoted += [
        f"edge label: |{label}|"
        for label in _EDGE_LABEL.findall(block)
        if not _is_quoted(label)
    ]
    unquoted += [
        f"subgraph title: {title}"
        for title in _SUBGRAPH.findall(block)
        # `subgraph id["Title"]` carries its quoting in the bracketed label.
        if not _is_quoted(title) and "[" not in title
    ]
    assert not unquoted, f"{path.name}: unquoted mermaid strings: {unquoted}"
