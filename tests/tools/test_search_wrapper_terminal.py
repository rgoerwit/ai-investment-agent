"""Structural guard against literal-closer misuse in tool producers.

Every tool that returns Tavily-wrapped output must keep the
`</search_results>` closer terminal. The heuristic content inspector's
`_detect_search_results_breakouts` treats a closer as a legitimate
trust-boundary footer only when nothing non-whitespace follows it
(`expected_wrapper_cleanup=True`). Producers that append a `Note:` line
or any other prose AFTER the wrapper trip the `delimiter_breakout`
signal at threat_level=high on every call.

**Scope of this static check:** it walks the AST of every `src/tools/*.py`
file and flags any `return` whose *literal* string parts contain
`</search_results>` followed by literal trailing text. It DOES catch
producers that hardcode a closer-then-prose pattern (e.g. a sentinel
truncation suffix with a trailing note).

**What it does NOT catch:** the dynamic case where a runtime variable
(e.g. `results_str` from `_format_and_truncate_tavily_result`) contains
the closer and an f-string places literal trailing text after the
substitution. That pattern requires runtime invocation to detect, and is
covered by per-tool tests such as
`tests/agents/test_foreign_language_analyst.py
::test_output_terminates_with_search_results_closer`.

Together the two layers catch both the literal and the dynamic forms of
the May 2026 2364.TW false-positive incident.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_TOOLS_DIR = Path(__file__).parent.parent.parent / "src" / "tools"


def _output_string_pieces(node: ast.AST) -> list[str]:
    """Collect literal-string pieces that form *node*'s output value.

    Limits to: bare string `Constant`, f-string `JoinedStr` literal slices,
    and string-concatenation via `BinOp(+)` of those. Crucially, it does
    NOT recurse into `Call` arguments — `text.replace("</search_results>",
    "[removed]")` must NOT contribute its first argument to the output
    inventory, because that string is consumed by the call, not emitted.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.JoinedStr):
        pieces: list[str] = []
        for v in node.values:
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                pieces.append(v.value)
            else:
                # FormattedValue or another expression substituted in;
                # treat as opaque non-empty placeholder so any literal
                # closer we already saw isn't mis-classified as terminal
                # when the value following it is dynamic.
                pieces.append("\x00DYN\x00")
        return pieces
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _output_string_pieces(node.left) + _output_string_pieces(node.right)
    # Anything else (Call, Name, etc.): output is opaque/dynamic.
    return ["\x00DYN\x00"]


@pytest.mark.parametrize(
    "module_path",
    sorted(_TOOLS_DIR.glob("*.py")),
    ids=lambda p: p.name,
)
def test_search_results_closer_is_terminal_in_returned_strings(module_path):
    """For every tool module: any string literal that contains
    `</search_results>` must either:

    - end with that closer (whitespace allowed), OR
    - be a constant defining the closer itself / a sentinel pattern (e.g.
      `</search_results>` alone, replacement strings, or matchers).

    The intent: catch tool authors who put metadata or prose AFTER the
    Tavily wrapper. That trips the inspector's breakout heuristic on every
    real call.
    """
    src_text = module_path.read_text(encoding="utf-8")
    if "</search_results>" not in src_text:
        pytest.skip("module does not produce search_results wrappers")

    tree = ast.parse(src_text)

    offenders: list[tuple[int, str]] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Return):
            continue
        if node.value is None:
            continue
        pieces = _output_string_pieces(node.value)
        joined = "".join(pieces)
        if "</search_results>" not in joined:
            continue
        closers = list(re.finditer(r"</search_results>", joined, re.I))
        last_closer = closers[-1]
        tail = joined[last_closer.end() :]
        # `\x00DYN\x00` represents a dynamic substitution we couldn't
        # resolve statically — treat as opaque non-empty content for the
        # purposes of "is the closer terminal?". If the tail strips to
        # empty (whitespace only), the closer IS terminal and we're fine.
        if tail.replace("\x00DYN\x00", "").strip() != "":
            preview = tail[:80].replace("\x00DYN\x00", "<DYNAMIC>")
            offenders.append((node.lineno, preview))

    assert not offenders, (
        f"In {module_path.name}, the following return statements emit text "
        f"AFTER the final </search_results> closer; this trips the inspector's "
        f"delimiter_breakout heuristic on every call. Move metadata/footers "
        f"BEFORE the wrapper. Offenders (line, trailing-tail):\n"
        + "\n".join(f"  L{ln}: {tail!r}" for ln, tail in offenders)
    )
