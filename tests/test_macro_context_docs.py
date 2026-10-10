import re
from pathlib import Path

_MERMAID_EDGE = re.compile(r"^\s*(\w+)(?:\[[^\]]*\])?\s*-\.?->")


def _mermaid_block(text: str) -> str:
    match = re.search(r"```mermaid\n(.*?)```", text, flags=re.DOTALL)
    assert match, "README architecture diagram is missing"
    return match.group(1)


def test_readme_mentions_macro_context_analyst_and_news_only_injection():
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "Macro Context Analyst" in readme
    assert "injected only into News Analyst" in readme

    diagram = _mermaid_block(readme)
    assert 'Macro["Pre-graph cached macro brief"] -.-> News["News Analyst"]' in diagram
    # News-only injection: the macro brief has no edge to any other seat.
    macro_edges = [
        line.strip()
        for line in diagram.splitlines()
        if (m := _MERMAID_EDGE.match(line)) and m.group(1) == "Macro"
    ]
    assert macro_edges == [
        'Macro["Pre-graph cached macro brief"] -.-> News["News Analyst"]'
    ]


def test_agentic_ai_101_mentions_pre_graph_macro_context():
    guide = Path("docs/AGENTIC-AI-101.md").read_text(encoding="utf-8")

    assert "Macro Context Analyst" in guide
    assert "[Pre-Graph Macro Context]" in guide
    assert "feeds it only to News Analyst" in guide
