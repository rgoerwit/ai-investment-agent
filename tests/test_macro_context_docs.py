from pathlib import Path


def test_readme_mentions_macro_context_analyst_and_news_only_injection():
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "Macro Context Analyst" in readme
    assert "MacroCtx -.-> NewsAnalyst" in readme
    assert "injects that background only into News Analyst in v1" in readme


def test_agentic_ai_101_mentions_pre_graph_macro_context():
    guide = Path("docs/AGENTIC-AI-101.md").read_text(encoding="utf-8")

    assert "Macro Context Analyst" in guide
    assert "[Pre-Graph Macro Context]" in guide
    assert "feeds it only to News Analyst" in guide
